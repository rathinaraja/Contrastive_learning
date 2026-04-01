# Training data path
folder_path_train = '/data_64T_1/Extracted_Tiles_Using_New_Method_1/P-0030/Informative_Part1' 

# Training parameters
batch_size_train = 16  # Reduced from original
num_workers = 4
tile_size = 256  # Slightly reduced image size
num_epochs = 1 

# DINO specific parameters
n_local_crops = 4  # Reduced from 6 to save memory
warmup_teacher_temp = 0.04
teacher_temp = 0.07
warmup_teacher_temp_epochs = 30
student_temp = 0.1
momentum_teacher = 0.996

# Training file
training_loss_file = "training_loss_dino.csv"

# Memory optimization
use_mixed_precision = True  # Enable mixed precision training
gradient_accumulation_steps = 2  # Accumulate gradients to simulate larger batch