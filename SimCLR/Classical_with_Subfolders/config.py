class Config:
    # Root folder containing all patient folders
    folder_path_train = '/data_64T_1/Extracted_Tiles_Using_New_Method_1'
    
    # Training parameters
    batch_size = 256
    num_workers = 4
    tile_size = 256
    num_epochs = 10
    
    # Model parameters
    temperature = 0.5
    triplet_weight = 1.0
    learning_rate = 3e-4
    
    # Output files
    training_loss_file = "training_loss_simclr.csv"
    
    # Memory optimization
    use_mixed_precision = True
    gradient_accumulation_steps = 2

