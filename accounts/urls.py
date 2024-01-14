from django.urls import path, include
from django.views.generic import TemplateView

from .views import SignUpView, activate, CustomPasswordResetView, CustomLoginView

urlpatterns = [
    path('signup/', SignUpView.as_view(), name="signup"),
    path('login/', CustomLoginView.as_view(), name="login"),
    path('password_reset/', CustomPasswordResetView.as_view(), name="password_reset"),
    path('activate/<uidb64>/<token>', activate, name="activate"),
    path('profile/', TemplateView.as_view(template_name="registration/profile.html"), name="profile"),

    path('', include('django.contrib.auth.urls')),
]
