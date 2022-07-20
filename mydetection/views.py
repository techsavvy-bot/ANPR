from datetime import datetime
from typing import final
from django.shortcuts import render
from itsdangerous import Serializer

from django.http.response import StreamingHttpResponse
from ANPR.anpr import numberplatedetection
# from streamApp.camera import VideoCamera

from mydetection.serializers import imageSerializer
from .models import image,numberplate

# from rest_framework.response import Response
from rest_framework.decorators import api_view

from difflib import SequenceMatcher

# @api_view(['GET'])
def home_request(request):
    number,photo=numberplatedetection() 
    # print(number," Number")
    users=numberplate.objects.all()
    print(users)
    maxi=0
    username=""
    finalNumber=""
    for user in users:
        # print()
        ratio=SequenceMatcher(None, number, user.number).ratio()
        
        if ratio>maxi:
            maxi=ratio
            username=user.name
            finalNumber=user.number
    
    if maxi<0.4:
        return render(request,'authenticate.html')

    print(username," ",finalNumber," ",maxi)
    userdetail=numberplate.objects.get(number=finalNumber)
    print(userdetail.name," ABCD")
    if userdetail.status==0:
        userdetail.timestampin=datetime.now()
        # userdetail.timestampout=None
        userdetail.status=1
        userdetail.save()
    else:
        userdetail.timestampout=datetime.now()
        userdetail.status=0
        userdetail.save()
    
    # qs=image.objects.all()
    # # print(qs)
    # serializer=imageSerializer(qs,many=True)
    # return Response(serializer.data,status=200)
    print(photo," Photo")
    return render(request,'home.html',context={"userdetail":userdetail,"photo":photo})
 
def logs(request):
    userdetail=numberplate.objects.filter(status=1)
    return render(request,'logs.html',context={"users":userdetail})