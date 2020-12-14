from django.contrib import admin
from django.urls import path, include
from keypoints.views import AnnotationView, FileUploadView, download
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('annotation', AnnotationView.as_view(), name='annotation'),
    path('', FileUploadView.as_view(), name='home'),
    path('<str:filepath>/', download, name='download')
]