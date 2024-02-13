import os
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import dataPipeline as DP

accepted=["pop","jazz","rock","folk","hiphop","punk","electronic"]
nums=[0,0,0,0,0,0,0]







def do(og):
    dest="E:\\fstMuse7dataSmall"
    for file in os.listdir(og):
        
        pathFile=os.path.join(og,file)
        id3 = ID3(pathFile)
        genre =DP.stdGenre( id3.get('TCON', ['Unknown'])[0])
        ind=accepted.index(genre)
        if nums[ind]!=100:
            nums[ind]+=1
            new_file_path = os.path.join(dest, file)
            os.rename(pathFile, new_file_path)
        if nums[0]>=100 and nums[1]>=100 and nums[2]>=100 and nums[3]>=100 and nums[4]>=100 and nums[5]>=100 and nums[6]>=100:
            return "done"





       # if genre in accepted:
        #    #file_name = os.path.basename(file)
         #   new_file_path = os.path.join(dest, file)
          #  os.rename(pathFile, new_file_path)
           # print(f"done {n}")


#og="E:\\fma_large"
#n=0
#for x in range(9,156):
 #   s=str(x)
  #  print(s)
   # if len(s)==1:
   #     pa=os.path.join(og,f"00{s}")
   # elif len(s)==2:
   #     pa=os.path.join(og,f"0{s}")
   # elif len(s)==3:
    #    pa=os.path.join(og,f"{s}")
    #do(pa,n)
og="E:\\fstMuse7data"
print(do(og))