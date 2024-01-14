from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm, UsernameField, AuthenticationForm, PasswordResetForm
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class CustomUserCreationForm(UserCreationForm):
    username = UsernameField(label="Username")
    email = forms.EmailField(label='Email Address')
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirm password', widget=forms.PasswordInput)

    class Meta:
        """Determines which fields are displayed. Username field does not need to be in the form therefore it's
        excluded."""
        model = User
        fields = ("username", "email", "password1", "password2")

    def clean_email(self):
        """Checks to see if the email already exists in the database"""
        email = self.cleaned_data['email'].lower()
        new = User.objects.filter(email=email)
        if new.count():
            raise ValidationError(" A user with that email already exists")

        return email

    def save(self, commit=True):
        """Saves the registered user using username/email/password. The username is the same as email at the moment
        but that may be changed at a later date."""
        user = User.objects.create_user(
            self.cleaned_data['username'],
            self.cleaned_data['email'],
            self.cleaned_data['password1']
        )
        user.is_active = False
        user.save()
        return user

class CustomAuthenticationForm(AuthenticationForm):
    error_messages = {
        "invalid_login": _(
            "Please enter a correct %(username)s and password. New accounts need to verify their emails before "
            "logging in."
        ),
        "inactive": _("This account is inactive."),
    }

class CustomPasswordResetForm(PasswordResetForm):
    """
    Custom password reset form created to change the label of the email field to "Email Address".
    """
    email = forms.EmailField(
        label=_("Email Address"),
        max_length=254,
        widget=forms.EmailInput(attrs={"autocomplete": "email"}),
    )
