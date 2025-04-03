# myapp/views.py
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.db import connection, IntegrityError
from .models import Employee, EmployeeLog  # Import EmployeeLog
from django.conf import settings
from django.core.files.storage import default_storage
from .mtcnn_logic import recognizer
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

def admin_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("admin_dashboard")
        else:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM Admin WHERE Username=%s AND Password=%s", [username, password])
                admin_user = cursor.fetchone()
            if admin_user:
                request.session["admin_user"] = username
                return redirect("admin_dashboard")
        messages.error(request, "Incorrect username or password.")
    return render(request, "adminlogin.html")

def add_employee(request):
    if request.method == "POST":
        name = request.POST.get("name")
        department = request.POST.get("department")
        employee_id = request.POST.get("employee_id")
        if not name or not department or not employee_id:
            messages.error(request, "All fields are required.")
            return redirect("add_employee")
        if Employee.objects.filter(employee_id=employee_id).exists():
            messages.error(request, "Employee ID already exists. Please use a unique ID.")
            return redirect("add_employee")
        try:
            employee = Employee(name=name, department=department, employee_id=employee_id)
            for i in range(1, 11):
                image_field = f"image{i}"
                if image_field in request.FILES:
                    setattr(employee, image_field, request.FILES[image_field])
            employee.save()
            messages.success(request, "Employee added successfully!")
            return redirect("admin_dashboard")
        except IntegrityError:
            messages.error(request, "Error saving employee. The Employee ID might already exist.")
            return redirect("add_employee")
    return render(request, "addemployee.html")

def admin_dashboard(request):
    employees = Employee.objects.all()
    employee_logs = EmployeeLog.objects.all().order_by('-date', '-time_in')  # Order by date and time_in descending
    return render(request, 'admins.html', {'employees': employees, 'employee_logs': employee_logs})

def delete_employee(request, employee_id):
    employee = get_object_or_404(Employee, id=employee_id)
    employee.delete()
    return redirect('admin_dashboard')

def edit_employee(request, employee_id):
    employee = get_object_or_404(Employee, id=employee_id)
    employee_images = {
        f"image{i}": request.build_absolute_uri(settings.MEDIA_URL + str(getattr(employee, f"image{i}")))
        for i in range(1, 11) if getattr(employee, f"image{i}", None)
    }
    if request.method == "POST":
        employee.name = request.POST["name"]
        employee.department = request.POST["department"]
        employee.employee_id = request.POST["employee_id"]
        removed_images = request.POST.get("removed_images", "").split(",")
        for image_field in removed_images:
            if hasattr(employee, image_field) and getattr(employee, image_field):
                image_path = getattr(employee, image_field).path
                if default_storage.exists(image_path):
                    default_storage.delete(image_path)
                setattr(employee, image_field, None)
        available_slots = [f"image{i}" for i in range(1, 11) if not getattr(employee, f"image{i}", None)]
        uploaded_files = request.FILES.getlist("new_images")
        for i, file in enumerate(uploaded_files):
            if i < len(available_slots):
                filename = default_storage.save(f"employees/{employee.id}/{file.name}", file)
                setattr(employee, available_slots[i], filename)
        employee.save()
        messages.success(request, "Employee updated successfully!")
        return redirect("admin_dashboard")
    return render(request, "editemployee.html", {"employee": employee, "employee_images": employee_images})

@csrf_exempt
def retrain_model(request):
    if request.method == "POST":
        try:
            # Reload the dataset and retrain the classifier
            recognizer.load_dataset()
            recognizer.update_classifier()
            return JsonResponse({"status": "success", "message": "Model retrained successfully"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Error retraining model: {str(e)}"})
    return JsonResponse({"status": "error", "message": "Invalid request method"})