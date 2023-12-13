train_maml()
│
├─ For each epoch:
│  ├─ Load tasks:
│  │  └─ For each task in num_tasks:
│  │     └─ task_loader()
│  │        ├─ load_images_from_folder(support)
│  │        └─ load_images_from_folder(query)
│  │
│  └─ maml_outer_loop():
│     └─ For each task in tasks:
│        ├─ maml_inner_loop():
│        │  ├─ For num_inner_updates:
│        │  │  ├─ Forward pass & compute loss (support set)
│        │  │  └─ Update weights
│        │  └─ Return updated_weights, support_set_loss
│        │
│        └─ Meta-optimization step:
│           ├─ Forward pass & compute loss (query set) with updated_weights
│           └─ Update meta-weights (reconstr_weight, smooth_weight, ssim_weight)
│
└─ End of training loop


train_maml()
│
├─ For each epoch:
│  ├─ Load tasks (Initialize tasks list):
│  │  └─ For each task in num_tasks:
│  │     ├─ task_loader():
│  │     │  ├─ Load support set:
│  │     │  │  └─ load_images_from_folder(support)
│  │     │  └─ Load query set:
│  │     │     └─ load_images_from_folder(query)
│  │     └─ Add to tasks list
│  │
│  └─ Execute MAML Outer Loop (maml_outer_loop):
│     └─ For each task in tasks list:
│        ├─ Execute MAML Inner Loop (maml_inner_loop):
│        │  ├─ For num_inner_updates:
│        │  │  ├─ Forward pass with support set:
│        │  │  │  └─ Model.build_inference_for_training(support_set_images)
│        │  │  ├─ Compute loss on support set:
│        │  │  │  └─ Model.build_loss()
│        │  │  └─ Update model weights based on loss
│        │  └─ Return updated_weights, support_set_loss
│        │                                       
│        ├─ Meta-Optimization Step:
│        │  ├─ Set updated weights in model
│        │  ├─ Forward pass with query set:
│        │  │  └─ Model.build_inference_for_training(query_set_images)
│        │  ├─ Compute loss on query set:
│        │  │  └─ Model.build_loss()
│        │  └─ Compute meta-gradients & update meta-weights
│        │
│        └─ Reset model to original weights
│
└─ Repeat for next epoch
