from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.views import PasswordResetView, LoginView
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils.encoding import force_str, force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.views.generic import CreateView

from accounts.forms import CustomUserCreationForm, CustomPasswordResetForm, CustomAuthenticationForm
from accounts.tokens import account_activation_token


# Create your views here.
class SignUpView(CreateView):
    """View to display for account registration. Requires email and password."""
    form_class = CustomUserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"

    def post(self, request, *args, **kwargs):
        """Processes the post request, sends an activation email if the account was successfully created."""
        response = super().post(request, *args, **kwargs)
        if self.object:
            if activate_email(request, self.object, self.object.email):
                messages.success(request,
                                 f"A verification email has been sent to {self.object.email}. Check your spam folder. "
                                 f"You will need to verify your email before logging in.")
                return response
            else:
                messages.error(request,
                               "Sign up failed. Unable to send verification email. Check if you typed your email address "
                               "correctly and try again.")
                return redirect("signup")
        else:
            return response


def activate_email(request, user, to_email):
    """Sends a pre-made account activation message using a template.
    :param request: The request object.
    :param user: The user object
    :param to_email: Email to send the activation token to.
    :return: 1 if the message was sent, 0 otherwise"""
    mail_subject = 'Activate your user account.'
    message = render_to_string('accounts/template_activate_account.html', {
        'user': user.username,
        'domain': get_current_site(request).domain,
        'uid': urlsafe_base64_encode(force_bytes(user.pk)),
        'token': account_activation_token.make_token(user),
        'protocol': 'https' if request.is_secure() else 'http'
    })
    email = EmailMessage(mail_subject, message, to=[to_email])

    return email.send()


def activate(request, uidb64, token):
    """Attempts to activate the user's account using information from the url.
    :param request: The request object.
    :param uidb64: URL encoded user id.
    :param token: Account activation token that needs verification.
    :return: Redirects to log in or home.
    """
    User = get_user_model()
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()

        messages.success(request, 'Thank you for your email confirmation. Now you can log into your account.')
        return redirect('login')
    else:
        messages.error(request, 'Activation link is invalid!')
        return redirect('home')


class CustomLoginView(LoginView):
    """Custom login view, the only change is the form which uses email/password to authenticate."""
    form_class = CustomAuthenticationForm


class CustomPasswordResetView(PasswordResetView):
    """Custom password reset view, the only change is the form which uses the label "Email Address" for the email
    field."""
    form_class = CustomPasswordResetForm
