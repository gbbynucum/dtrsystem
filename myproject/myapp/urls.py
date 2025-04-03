# myapp/urls.py
from django.urls import path
from .views import admin_login, add_employee, admin_dashboard, delete_employee, edit_employee, retrain_model
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin-login/", admin_login, name="admin_login"),
    path("add-employee/", add_employee, name="add_employee"),
    path('admin_dashboard/', admin_dashboard, name='admin_dashboard'),
    path('delete_employee/<int:employee_id>/', delete_employee, name='delete_employee'),
    path("edit_employee/<int:employee_id>/", edit_employee, name="edit_employee"),
    path("retrain-model/", retrain_model, name="retrain_model"),  # New URL
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)