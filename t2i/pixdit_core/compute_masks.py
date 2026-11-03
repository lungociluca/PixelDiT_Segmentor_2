import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops

import t2i.config as local_config


def get_gt_bin(gt, label_ids, h, w):
    gt = gt[gt != 255]
    gt_binary = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=local_config.label_count+1)
    gt_binary = F.interpolate(gt_binary.permute(2,0,1).unsqueeze(0).to(torch.bfloat16), (h, w), mode="nearest")[0]
    gt_binary =  einops.rearrange(gt_binary, "b h w -> b (h w)")
    return torch.stack([gt_binary[lid] for lid in label_ids], dim=0)


# def save_intermediary_plots(merged_mask, gt_bin, gt_ids, h, w):
#     import matplotlib.pyplot as plt
#     import os
#     if os.path.isfile("print"):
#         with torch.no_grad():
#             for ii in range(gt_bin.shape[0]):
#                 pred = merged_mask[ii].detach().float().cpu().view(h,w)
#                 gt = gt_bin[ii].detach().float().cpu().view(h,w)

#                 fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#                 axes[0].imshow(pred, cmap="hot")
#                 axes[0].set_title(f"pred of {gt_ids[ii]}")
#                 axes[0].axis("off")

#                 axes[1].imshow(gt, cmap="hot",)
#                 axes[1].set_title("gt_bin")
#                 axes[1].axis("off")

#                 plt.tight_layout()
#                 plt.savefig(f"trash/{len(os.listdir('trash'))}.jpg")
#                 plt.close(fig)
#         # os.remove("print")


def min_max(masks, dim):
    # reduce over the last dim, keep dims for broadcasting
    mins = masks.amin(dim=dim, keepdim=True)
    maxs = masks.amax(dim=dim, keepdim=True)
    return (masks - mins) / (maxs - mins + 1e-8)

def clasify_masks(visual_tokens, language_tokens, language_weights, language_no_object, label_ids):
    # WARNING: language_tokens are kept on CPU
    # visual_tokens -> (l, d), language_tokens -> (t,d)
    visual_tokens = visual_tokens.sum(dim=1) # sum across layers TODO: try other ways of aggregating visual features for comparison with language
    language_tokens_selected = language_tokens[:171].to(local_config.device)
    language_targets = language_weights * language_tokens_selected
    language_targets = language_targets.sum(dim=-2)
    language_targets = torch.cat([language_targets, language_no_object[None,:,:]], dim=0).permute(0,2,1)
    my_norm = lambda x: torch.nn.functional.normalize(x)
    classes = my_norm(visual_tokens) @ my_norm(language_targets.to(torch.bfloat16))
    return classes


def compute_masks(features, tokens, vl_features, language_tokens, learnable_token_norm_weights, language_weights, language_no_object, label_ids):
    B, L, P, D = features.shape
    my_norm = lambda x: x if not local_config.norm_before_dot else torch.nn.functional.normalize(x, dim=-1)
    # min_max = lambda x: (x - x.min()) / (x.max() - x.min())
    masks = my_norm(features) @ my_norm(tokens).permute(0, 1, 3, 2)

    # masks (b, l, p, t) * (b, l, 1, t)
    masks = learnable_token_norm_weights(masks)
    masks = masks.sum(dim=1)

    mask_classes = clasify_masks(vl_features, language_tokens, language_weights, language_no_object, label_ids)
    if local_config.save_learned_tokens:
        return masks, mask_classes.squeeze(-1).unsqueeze(0)
    else:
        masks = masks.sigmoid()
        mask_classes = F.softmax(mask_classes.squeeze(-1).unsqueeze(0), dim=-1)[:,:-1,:]
        print(mask_classes.permute(0,2,1).shape, masks.permute(0,2,1).shape)
        semseg = torch.einsum("bqc,bqp->bcp", mask_classes.permute(0,2,1), masks.permute(0,2,1))
        return semseg


def compute_loss(features, tokens, vl_features, language_tokens,
                 learnable_token_norm_weights, language_weights, language_no_object, gt, label_ids, h, w, criterion):
    B, L, P, D = features.shape
    # gt_bin = get_gt_bin(gt, label_ids, h, w).to(local_config.device)

    mask_logits, mask_classes_logits = compute_masks(features, tokens, vl_features,
                                              language_tokens, learnable_token_norm_weights,
                                              language_weights, language_no_object, label_ids)

    mask_logits = mask_logits.permute(0,2,1).view(1, local_config.no_vision_learnable_tokens, h, w)
    mask_classes_logits = mask_classes_logits.permute(0,2,1)

    outputs = {
        "pred_logits": mask_classes_logits,   # [B, Q, C+1]   last col = no-object
        "pred_masks":  mask_logits,    # [B, Q, Hm, Wm]
    }

    present = torch.unique(gt)
    present = present[present != 255]               # drop ignore

    masks  = torch.stack([(gt == c) for c in present]).float().to(local_config.device)  # [T, Hg, Wg], {0,1}

    targets = [
        {"labels": torch.LongTensor(label_ids).to(local_config.device),        # [T_b]  long, ids into the C vocabulary
         "masks":  masks}         # [T_b, Hg, Wg]  {0,1}, padding masks removed
    ]
    losses = criterion(outputs, targets)
    match_indicies = [i.item() for i in criterion.last_indicies[0][0]]

    # with open("debug_overfit.txt", "a") as f:
    #     f.write(f"target labels: {torch.LongTensor(label_ids)}\ntarget inds source: {[i.item() for i in criterion.last_indicies[0][0]]}, gt: {[i.item() for i in criterion.last_indicies[0][1]]}\npred_logits:\n")
    #     for matched_idx in match_indicies:
    #         f.write(f"{matched_idx} ------>\t{torch.argmax(F.softmax(mask_classes_logits[0,matched_idx,:] // 0.01))}\n")
    #     f.write("\n\n") 

    #     import json
    #     with open("catseg_configs/coco.json") as f:
    #         clean_labels = [x for x in json.load(f) if x != "unknown"] + ["empty"]
    #     ii = 0
    #     for pid, gid in zip(match_indicies, [i.item() for i in criterion.last_indicies[0][1]]):
    #         import os
    #         pred_label = torch.argmax(F.softmax(mask_classes_logits[0,pid,:] // 0.01))
    #         try:
    #             current_dir = f"trash/{clean_labels[pred_label]}"
    #         except IndexError as e:
    #             print(pred_label, len(clean_labels))
    #             raise(e)

    #         os.makedirs(current_dir, exist_ok=True)
    #         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #         axes[0].imshow(mask_logits[0, pid].detach().cpu(), cmap="hot")
    #         axes[0].set_title(f"pred with index {pid}")
    #         axes[0].axis("off")

    #         axes[1].imshow(masks[gid].detach().cpu(), cmap="hot",)
    #         axes[1].set_title("gt_bin")
    #         axes[1].axis("off")

    #         plt.tight_layout()
    #         plt.savefig(f"{current_dir}/{len(current_dir)}_{ii}.jpg")
    #         plt.close(fig)
            
    #         ii += 1
    #         # if ii == 2:
    #         #     break

    predictions = torch.stack([mask_logits[:,i] for i in match_indicies], dim=1)
    return predictions, sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict), losses["loss_dice"]
