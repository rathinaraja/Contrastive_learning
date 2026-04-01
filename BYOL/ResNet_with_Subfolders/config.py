folder_path_train = '/data_64T_1/Extracted_Tiles_Using_New_Method_1'    
subfolder_names = ['Informative_Part1']
batch_size_train = 256  # Reduced batch size due to twin networks
num_workers = 4
tile_size = 256 
num_epochs = 100
training_loss_file = "training_loss_byol.csv"