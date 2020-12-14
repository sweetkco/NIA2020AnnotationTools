from django.db import models

# Create your models here.
class ImageDataset(models.Model):
    img_no = models.AutoField(primary_key=True)
    img_path = models.CharField(null=False, max_length=200)
    frame_no = models.IntegerField(null=False)
    camera_no = models.IntegerField(null=False)
    joint_2d = models.CharField(null=True, max_length=400)
    joint_3d = models.CharField(null=True, max_length=2000)
    intrinsics = models.CharField(null=True, max_length=150)
    extrinsics = models.CharField(null=True, max_length=250)
    rotation = models.CharField(null=True, max_length=1700)
    shape_params = models.CharField(null=True, max_length=300)
    trans_params = models.CharField(null=True, max_length=150)
    width = models.IntegerField(null=True)
    height = models.IntegerField(null=True)

class VideoDataset(models.Model):
    camera_no = models.IntegerField()
    file = models.FileField(upload_to='data')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    intrinsics = models.CharField(null=True, max_length=150)
    extrinsics = models.CharField(null=True, max_length=250)

