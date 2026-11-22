"""Self-predicted-pose rollout of the released DrivingWorld modules (one batch = one window's K samples).

Everything model-side is the released code at the pinned commit (models/model.py TrainTransformers, models/ar.py GPT,
modules/tokenizers/*), called the way tools/test_long_term_demo.py calls it. The kit adds only what the release lacks:
  1. the model's own next pose: the released inference (generate_gt_pose_gt_yaw) is only given the next pose, while the
     paper's next-state prediction samples yaw, then x, then y, then the image tokens. The yaw and x steps call the
     released GPT.eval_third unchanged; the y step needs the training head pose_y_head (GPT.forward), but eval_third
     applies pose_x_head at that position (never exercised by the released demos), so internal_hidden() repeats
     eval_third line for line up to its heads and pose_y_head is applied to position 2;
  2. per-sample random streams: the released top_k_sampling draws from the global generator; topk_rows() keeps its
     arithmetic and draws each row from that row's own torch.Generator (identical draws for a batch of one);
  3. batch decoding: the released Tokenizer.z_to_clips decodes only the last frame of the flattened batch, so each row is
     decoded by its own z_to_clips call.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def topk_rows(logits, top_k, temperature, generators):
    """utils/generat_utils.top_k_sampling with one generator per row.

    Same arithmetic as released: logits / T, threshold = the k-th largest UNtempered logit, -inf below it, softmax,
    multinomial. masked_fill replaces the boolean-index assignment (same values, no host sync)."""
    if logits.ndim != 3 or logits.shape[1] != 1 or logits.shape[0] != len(generators):
        raise ValueError('logits [B, 1, N] with one generator per row required')
    logits_topk = logits / temperature
    v, _ = torch.topk(logits, min(top_k, logits_topk.size(-1)))
    logits_topk = logits_topk.masked_fill(logits_topk < v[:, :, [-1]], -float('Inf'))
    probs = F.softmax(logits_topk, dim=-1)
    return torch.cat([torch.multinomial(probs[b:b + 1, 0, :], num_samples=1, generator=g)
                      for b, g in enumerate(generators)], dim=0)


def sample(logits, cfg, generators):
    if cfg['method'] != 'top_k':
        raise ValueError('registered sampler is top_k')
    return topk_rows(logits, cfg['top_k'], cfg['temperature'], generators)


def internal_hidden(gpt, ar_emb, yaws=None, poses=None, new_tokens=None, drop_flag=False):
    """models/ar.py GPT.eval_third up to (not including) its output heads, line for line."""
    mask_spatial_curr, mask_ar_curr = gpt.organize_attn_mask(drop_flag)
    B, L1, C = ar_emb.shape
    sos_embeddings = gpt.sos_emb.repeat((B, 1, 1))
    if (yaws is not None) and (poses is None) and (new_tokens is None):
        yaw_emb, _, _ = gpt.get_yaw_pose_emb(None, yaws)
        yaw_embeddings = gpt.yaw_projector(yaw_emb[:, :, 0, :])
        intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings], dim=1)
    elif (yaws is not None) and (poses is not None) and (new_tokens is None):
        yaw_emb, pose_x_emb, pose_y_emb = gpt.get_yaw_pose_emb(poses, yaws)
        yaw_embeddings = gpt.yaw_projector(yaw_emb[:, :, 0, :])
        pose_x_embeddings = gpt.pose_x_projector(pose_x_emb[:, :, 0, :])
        if poses.shape[2] == 2:
            pose_y_embeddings = gpt.pose_y_projector(pose_y_emb[:, :, 0, :])
            intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings, pose_x_embeddings, pose_y_embeddings], dim=1)
        else:
            intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings, pose_x_embeddings], dim=1)
    elif (yaws is not None) and (poses is not None) and (new_tokens is not None):
        yaw_emb, pose_x_emb, pose_y_emb = gpt.get_yaw_pose_emb(poses, yaws)
        yaw_embeddings = gpt.yaw_projector(yaw_emb[:, :, 0, :])
        pose_x_embeddings = gpt.pose_x_projector(pose_x_emb[:, :, 0, :])
        pose_y_embeddings = gpt.pose_y_projector(pose_y_emb[:, :, 0, :])
        new_embeddings = gpt.img_projector(new_tokens)
        intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings, pose_x_embeddings, pose_y_embeddings, new_embeddings], dim=1)
    else:
        intra_query_embeddings = sos_embeddings
    _, L, _ = intra_query_embeddings.shape
    assert L == L1
    h = intra_query_embeddings + ar_emb
    for i in range(gpt.auto_regressive_num):
        h = gpt.causal_space_blocks[i](h, mask_ar_curr[:, :L1, :L1])
    return h


def pose_y_logits(gpt, ar_emb3, yaws, x_tokens, drop_flag):
    """Logits of the y token with the training head (GPT.forward: pose_y_head at position 2)."""
    h = internal_hidden(gpt, ar_emb3, yaws=yaws, poses=x_tokens, drop_flag=drop_flag)
    return gpt.pose_y_head(h[:, gpt.yaw_token_size + 1:gpt.yaw_token_size + gpt.pose_token_size, :])


def image_tokens(tw, intra, yaws, poses, drop_flag, codebook, generators, cfg):
    """The image-token loop of models/model.py generate_gt_pose_gt_yaw with the per-row sampler."""
    gpt = tw.model
    new_tokens = None
    predict_indices = []
    for i in range(tw.img_token_size):
        logits = gpt.eval_third(intra[:, :i + tw.yaw_token_size + tw.pose_token_size + 1, :], yaws=yaws, poses=poses,
                                new_tokens=new_tokens, drop_flag=drop_flag)['img_logits']
        logits = logits[:, i:i + 1, :]
        idx_next = sample(logits, cfg, generators)
        predict_indices.append(idx_next)
        tmp_tokens = codebook(idx_next)
        new_tokens = tmp_tokens if new_tokens is None else torch.cat([new_tokens, tmp_tokens], dim=1)
    predict_indices = torch.cat(predict_indices, dim=1)
    return rearrange(predict_indices, 'b (h w) -> b h w', h=tw.latent_size[0], w=tw.latent_size[1])


def sample_next_state(tw, feature, pose_win, yaw_win, codebook, generators, pose_cfg, image_cfg, given=None):
    """One next state for every row: yaw, x, y (own prediction, or `given` = (yaw, x, y) index tensors), then 512 tokens.

    pose_win (B, 15, 2) and yaw_win (B, 15, 1) are continuous relative poses (first entry zeroed by the caller)."""
    from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices
    gpt = tw.model
    # All flags are False here; a CPU copy gives organize_attn_mask the same masks without a device sync per call.
    drop_flag = torch.isinf(pose_win[:, 0, 0]).cpu()
    pose_indices = poses_to_indices(pose_win, tw.pose_x_vocab_size, tw.pose_y_vocab_size)
    yaw_indices = yaws_to_indices(yaw_win, tw.yaw_vocab_size)
    intra = gpt.eval_first_second(feature, pose_indices, yaw_indices, drop_flag)
    ny = tw.yaw_token_size
    if given is None:
        logits = gpt.eval_third(intra[:, :ny, :], drop_flag=drop_flag)['yaw_logits'][:, 0:1, :]
        yaw_tok = sample(logits, pose_cfg, generators)
        logits = gpt.eval_third(intra[:, :ny + 1, :], yaws=yaw_tok[:, :, None], drop_flag=drop_flag)['pose_x_logits'][:, 0:1, :]
        x_tok = sample(logits, pose_cfg, generators)
        logits = pose_y_logits(gpt, intra[:, :ny + 2, :], yaw_tok[:, :, None], x_tok[:, :, None], drop_flag)
        y_tok = sample(logits, pose_cfg, generators)
    else:
        yaw_tok, x_tok, y_tok = given
    yaws = yaw_tok[:, :, None]                                   # (B, 1, 1) as yaw_gt_indices[:, 0:1, :]
    poses = torch.stack([x_tok, y_tok], dim=-1)                  # (B, 1, 2) as pose_gt_indices[:, 0:1, :]
    img = image_tokens(tw, intra, yaws, poses, drop_flag, codebook, generators, image_cfg)
    return yaw_tok, x_tok, y_tok, img


def decode_rows(tokenizer, window_tokens, new_tokens):
    """Released Tokenizer.z_to_clips per row on the 16-frame token window (15 previous + new), as the demo builds it.

    window_tokens (B, 15, 512), new_tokens (B, 16, 32) -> uint8 frames (B, 256, 512, 3) by the demo's truncation rule."""
    import numpy as np
    h, w = new_tokens.shape[-2:]
    frames = []
    for b in range(window_tokens.shape[0]):
        condition_tokens = rearrange(window_tokens[b:b + 1], 'b F (h w) -> b F h w', h=h)
        sliding = rearrange(torch.cat([condition_tokens, new_tokens[b:b + 1, None]], dim=1), 'b F h w -> (b F) h w', F=condition_tokens.shape[1] + 1)
        img = tokenizer.z_to_clips(sliding).cpu()
        frames.append((img[0].permute(1, 2, 0).numpy() * 255).astype('uint8'))
    return np.stack(frames)


def rollout(tw, tokenizer, cond_tokens, cond_xy, cond_yaw, seeds, frames, pose_cfg, image_cfg, autocast=True,
            on_frame=None, given_poses=None):
    """Autoregressive 10 Hz rollout of len(seeds) samples from one conditioning window.

    cond_tokens (1, 15, 512) long; cond_xy (15, 2), cond_yaw (15, 1) relative poses of the conditioning frames.
    Returns dict(tokens (B, frames, 512), pose_tokens (B, frames, 3) [yaw, x, y], pose_values (B, frames, 3)
    [yaw deg, x m, y m], frames uint8 (B, frames, 256, 512, 3))."""
    import numpy as np
    from modules.tokenizers.pose_tokenizer import indices_to_pose, indices_to_yaws
    device = cond_tokens.device
    B = len(seeds)
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    codebook = tokenizer.vq_model.quantize.embedding
    start_token = cond_tokens.repeat(B, 1, 1)
    pose_win = torch.as_tensor(np.asarray(cond_xy), dtype=torch.float32, device=device)[None].repeat(B, 1, 1)
    yaw_win = torch.as_tensor(np.asarray(cond_yaw), dtype=torch.float32, device=device)[None].repeat(B, 1, 1)
    tokens, pose_tokens, pose_values, images = [], [], [], []
    for t in range(frames):
        pose_win[:, 0, :] = 0.0                                    # the demos zero the window's first relative pose
        yaw_win[:, 0, :] = 0.0
        start_feature = codebook(start_token)                      # no l2 norm, as the demos
        given = None if given_poses is None else tuple(x[:, t:t + 1] for x in given_poses)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            yaw_tok, x_tok, y_tok, img = sample_next_state(tw, start_feature, pose_win, yaw_win, codebook, generators,
                                                           pose_cfg, image_cfg, given=given)
        x, y = indices_to_pose(x_tok, y_tok, tw.pose_x_vocab_size, tw.pose_y_vocab_size)
        yaw = indices_to_yaws(yaw_tok, tw.yaw_vocab_size)
        images.append(decode_rows(tokenizer, start_token, img))
        flat = rearrange(img, 'b h w -> b 1 (h w)')
        start_token = torch.cat([start_token[:, 1:], flat], dim=1)
        pose_win = torch.cat([pose_win[:, 1:], torch.stack([x, y], dim=-1).float()], dim=1)
        yaw_win = torch.cat([yaw_win[:, 1:], yaw[:, :, None].float()], dim=1)
        tokens.append(flat[:, 0].cpu())
        pose_tokens.append(torch.cat([yaw_tok, x_tok, y_tok], dim=1).cpu())
        pose_values.append(torch.cat([yaw, x, y], dim=1).float().cpu())
        if on_frame is not None:
            on_frame(t)
    return dict(tokens=torch.stack(tokens, 1).numpy(), pose_tokens=torch.stack(pose_tokens, 1).numpy(),
                pose_values=torch.stack(pose_values, 1).numpy(), frames=np.stack(images, 1))


def integrate(pose_values):
    """SE(2) integration of relative (yaw deg, x m, y m) steps; returns (N, 3) [x m, y m, heading deg] in the t0 ego frame."""
    import numpy as np
    out, x, y, th = [], 0.0, 0.0, 0.0
    for yaw, dx, dy in np.asarray(pose_values, np.float64):
        c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
        x, y, th = x + c * dx - s * dy, y + s * dx + c * dy, th + yaw
        out.append([x, y, th])
    return np.array(out)
