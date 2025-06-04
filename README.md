# Senior Research Project: Applying Cartoony Style Transfer to Realistic Videos
The purpose of this project was to create a tool to make the cartoon animation process quicker and less time consuming. Through the use of generative adversarial neural networks, style transfer is applied to realistic videos of people to convert them into cartoon people in the style of old Disney movies. 

The training datasets were too large to upload to GitHub, but they can be found in the Google Drive folder instead. Download all the datasets to be used in training later. If you would like to use and process your own training data, follow the process below:

1. Open folder 00_DATA_PROCESSING. Open file make_inds.py and change variable globname to the file path to your unprocessed content data. Run the file to get a list of random training and testing indices.
2. Open file content_process.py and change variable globname to the file path to your unprocessed content data. Create folders for your train and test data, and change the variable newfname to the file path of these folders. Run the file to get your processed content data. 
3. Open file style_process.py and change variable globname to the file path to your unprocessed style data. Change variable newfname to the file path and desired file name of your processed style data. Run the file to get your processed style data.
4. Open file blurry_style_process.py and change variable globname to the file path of your processed style data. Change variable fname to the file path of the desired folder and file name of your processed blurry data. Run the file to get your processed blurry style data. 

Training with individual images (content dataset):
1. Open folder 01_TRAIN_INDIVIDUAL. Change the variables in file config.py to match your intended file and directory locations. Weights can be altered in config.py too, as well as whether or not you are in the initialization phase of training. 
2. Open and run file train.py to begin training. The current epoch’s progress will be displayed in the terminal, and the network weights will be saved after each epoch if you have indicated saving in config.py.

Training with videos: 
1. Download the video you want to apply style transfer to and save it somewhere you can access.
2. Open folder 02_TRAIN_VIDEO. Open and run file combined_vid_transfer.py and follow the instructions displayed in the terminal. 
Run file join_frames.py to create the video from the frames in the directory specified by join_frames.py’s variable path. 
