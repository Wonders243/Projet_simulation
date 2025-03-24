from django.db import models

# Create your models here.

class Point(models.Model):
    x = models.IntegerField()
    y = models.IntegerField()
