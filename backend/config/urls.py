from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)
from rest_framework.permissions import AllowAny

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("core.urls")),
    path("api/trading/", include("trading.urls")),
    # Swagger/OpenAPI documentation (без авторизации)
    path("api/schema/", SpectacularAPIView.as_view(permission_classes=[AllowAny]), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema", permission_classes=[AllowAny]), name="swagger-ui"),
    path("api/redoc/", SpectacularRedocView.as_view(url_name="schema", permission_classes=[AllowAny]), name="redoc"),
]

