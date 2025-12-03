from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from core import views

urlpatterns = [
    path("", views.api_root, name="api-root"),
    path("health/", views.health, name="health"),
    path("auth/register/", views.RegisterView.as_view(), name="auth-register"),
    path("auth/login/", views.CustomTokenObtainPairView.as_view(), name="auth-login"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="auth-refresh"),
    path("auth/me/", views.ProfileView.as_view(), name="auth-me"),
]

