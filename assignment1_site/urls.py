from django.contrib import admin
from django.urls import path
from analysis.views import results_page

urlpatterns = [
    path("", results_page, name="results"),
    path("admin/", admin.site.urls),
]