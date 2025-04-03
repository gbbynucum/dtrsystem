# myapp/models.py
from django.db import models
from django.contrib.auth.hashers import make_password, check_password

class Admin(models.Model):
    """Custom Admin model with password hashing"""
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=255)

    def save(self, *args, **kwargs):
        """Hashes the password before saving to the database."""
        self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def check_password(self, raw_password):
        """Checks if the entered password matches the stored hashed password."""
        return check_password(raw_password, self.password)

    def __str__(self):
        return self.username


class Employee(models.Model):
    name = models.CharField(max_length=255)
    department = models.CharField(max_length=255)
    employee_id = models.CharField(max_length=50, unique=True)
    date = models.DateField(auto_now_add=True)

    # Image fields (up to 10)
    image1 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image2 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image3 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image4 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image5 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image6 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image7 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image8 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image9 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image10 = models.ImageField(upload_to='employees/', null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.employee_id})"


class EmployeeLog(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='logs')
    time_in = models.DateTimeField(null=True, blank=True)
    time_out = models.DateTimeField(null=True, blank=True)
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.employee.name} - {self.date}"