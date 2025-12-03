from django.http import JsonResponse
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from core.serializers import RegisterSerializer, UserSerializer


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["email"] = user.email
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data["user"] = UserSerializer(self.user).data
        return data


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


class RegisterView(APIView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "user": UserSerializer(user).data,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            },
            status=status.HTTP_201_CREATED,
        )


class ProfileView(APIView):
    def get(self, request):
        return Response(UserSerializer(request.user).data)


def health(_request):
    return JsonResponse({"status": "ok"})


def api_root(_request):
    """Корневой эндпоинт API"""
    return JsonResponse(
        {
            "name": "Photon Trading API",
            "version": "1.0.0",
            "status": "ok",
            "endpoints": {
                "health": "/api/health/",
                "auth": {
                    "register": "/api/auth/register/",
                    "login": "/api/auth/login/",
                    "refresh": "/api/auth/refresh/",
                    "me": "/api/auth/me/",
                },
                "trading": {
                    "symbols": "/api/trading/symbols/",
                    "market_data": "/api/trading/market-data/",
                    "decisions": "/api/trading/decisions/",
                    "agents": {
                        "market_monitor": "/api/trading/agents/market-monitor/",
                        "decision_maker": "/api/trading/agents/decision-maker/",
                        "execution": "/api/trading/agents/execution/",
                    },
                },
            },
        }
    )

