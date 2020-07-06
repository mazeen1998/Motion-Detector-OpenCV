##Motion detection
import cv2,datetime,pandas
import numpy as np
time=[]
s=[None,None]
df=pandas.DataFrame(columns=['Start',"End"])
fframe=None
i=0
v=cv2.VideoCapture(0)
while True:
	c,f=v.read()
	g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
	g1=cv2.GaussianBlur(g,(21,21),0)
	status=0
	if fframe is None:
		fframe=g1
		continue
	
	delta=cv2.absdiff(g1,fframe)
	status=0
	# delta=cv2.absdiff(delta,g1)
	threshd=cv2.threshold(delta,25,255,cv2.THRESH_BINARY)[1] ###o/p two vales retval,threshold
	# print(cv2.threshold(delta,30,255,cv2.THRESH_BINARY)[1])
	# at=cv2.adaptiveThreshold(delta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	# ot=cv2.threshold(delta,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	t=cv2.dilate(threshd,None,iterations=1)
	# if t1 is None:
	# 	t1=t
	# 	continue
	# print(t)
	# # print(type(t))
	# # print(t.shape)
	# print(t1)
	# # print(type(t1))
	# # print(t.shape)
	# if t is t1:
	# 	status=0
	# if t is not t1:
	# 	status=1
	# print(cv2.findContours(t.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE))
	cotur,_=cv2.findContours(t.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for c in cotur:
		if cv2.contourArea(c)<1000:###eleminates minimal difference
			continue
		if cv2.contourArea(c)>8000:
			continue
		(x,y,w,h)=cv2.boundingRect(c)
		cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),3)
		# cv2.drawContours(f,c,-1,(255,0,0),3)
		status=1
	print(status)
	s.append(status)
	print(s)
	s=s[-2:]
	# print(s)
	if s[-1]==1 and s[-2]==0:
		time.append(datetime.datetime.now())
		i+=1
		cv2.imwrite('Image{}.jpg'.format(i),f)
	if s[-1]==0 and s[-2]==1:
		time.append(datetime.datetime.now())



	# cv2.imshow('contour',cotur[0])
	# cv2.imshow('fframe',fframe)
	cv2.imshow('frame',f)
	cv2.imshow('delta-thres',np.hstack((delta,t)))
	# cv2.imshow('diff',t)
	# cv2.imshow('thres',t)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
# print(cotur)
print(s)
print(time)
for i in range(0,len(time),2):
	df=df.append({'Start':time[i],'End':time[i+1]},ignore_index=True)
df.to_csv('time3.csv')
v.release()
cv2.destroyAllWindows()
