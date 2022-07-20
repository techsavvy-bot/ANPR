from datetime import datetime
from django.db import models

# Create your models here.
class image(models.Model):
    number=models.IntegerField(default=0)
    numberimage=models.ImageField()

class numberplate(models.Model):
    name=models.TextField()
    number=models.TextField()
    status=models.IntegerField(default=0)
    timestampin=models.DateTimeField(default=datetime.now,null=True, blank=True)
    timestampout=models.DateTimeField(default=datetime.now,null=True, blank=True)
