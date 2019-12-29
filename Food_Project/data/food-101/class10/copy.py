import os
import shutil

foldernames=['apple_pie', 'baklava', 'beef_tartare', 'beignets', 'bread_pudding', 'baby_back_ribs', 'beef_carpaccio', 'beet_salad', 'bibimbap', 'breakfast_burrito']

for foldername in foldernames:
    os.mkdir("./train/" + foldername)
    os.mkdir("./test/" + foldername)
    i=1
    for filename in os.listdir("./images/" + foldername):
        dir_dst=""
        if(i<150):
            dir_dst="./train/"
        elif(i<250):
            dir_dst="./test/"
        shutil.copy("./images/" + foldername + "/" + filename, dir_dst + foldername)
        i=i+1
    print(filename)
