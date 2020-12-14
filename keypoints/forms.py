from django import forms
from keypoints.models import VideoDataset

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = VideoDataset
        fields = ('camera_no', 'file', 'intrinsics', 'extrinsics')
