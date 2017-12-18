# Face_verification_lcx
	A demo for face verification based on MTCNN and finetuned squeezenet
	Detection part,MTCNN from https://github.com/DuinoDu/mtcnn
	Verification part,the orignal squeezenet is from https://github.com/DeepScale/SqueezeNet
	We use lfw dataset to finetune

# Requirement
	python2 or python3
	numpy,caffe,opencv

# License

```
This code is distributed under MIT LICENSE and BSD LICENSE
```


# Use this demo

	This demo can detect face about 1.5m away, modify 'scale' in register.py to change the max detection distance 
	(eg.scale->5 about 1.5m
	 scale->2 about 4m)
	
	You should use the demo in this order:
	1.Run register.py
	2.Register faces(Save cropped faces)  
		Manually
		   Press 'r',and follow the instructions,input number corresponding to the face,and then input name.
		   To quit this mode,register all the faces in the frame,or input end.
		   
		Automatically(only for 3 faces always in frame)
	       You should press 'r' to register 3 faces manually,and make sure the 3 faces are aways inside
		   and can be detected before press 's' to register automatically .


		The registered faces pictures are in folder /capture.You can delete or add pictures to subfolders of /capture.

	3.Stop reigister.py(press 'e') and run it again

	4.Press 'l' to load pictures and get vectors.

    5.Press 'c' to calculate the thresholds.

	6.Now you can see faces and names,you can press 'e' to stop.
​	