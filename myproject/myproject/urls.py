# myproject/urls.py
from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.homepage, name='home'),
    path('admins/', views.adminpage, name='admin_dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('addemployee/', views.add_employee, name='add_employee'),
    path('adminlogin/', views.admin_login, name='admin_login'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_employee_data/', views.get_employee_data, name='get_employee_data'),
    path('log_attendance/', views.log_attendance, name='log_attendance'),  # New endpoint
    path('', include('myapp.urls')),
]