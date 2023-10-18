

# Optimizer & LR scheme
  ngpus=8,
  batch_size=128,  # per GPU

  epochs=600, 
  opt='sgd',  
  momentum=0.9,

  lr=0.5, 
  lr_scheduler='cosineannealinglr', 
  lr_warmup_epochs=5, 
  lr_warmup_method='linear', 
  lr_warmup_decay=0.01, 


  # Regularization and Augmentation
  weight_decay=2e-05, 
  norm_weight_decay=0.0,

  label_smoothing=0.1, 
  mixup_alpha=0.2, 
  cutmix_alpha=1.0, 
  auto_augment='ta_wide', 
  random_erase=0.1, 
  
  ra_sampler=True,
  ra_reps=4,


  # EMA configuration
  model_ema=True, 
  model_ema_steps=32, 
  model_ema_decay=0.99998, 


  # Resizing
  interpolation='bilinear', 
  val_resize_size=232, 
  val_crop_size=224, 
  train_crop_size=176,
