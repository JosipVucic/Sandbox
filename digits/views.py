from django.contrib import messages
from django.contrib.auth.mixins import UserPassesTestMixin, LoginRequiredMixin
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import FormView

from .forms import ImageForm
from .neural.model import GACNN
from .neural.preprocessing import preprocess_image


class HomeView(LoginRequiredMixin, FormView):
    """
    The home view, allows for image upload if the user is authenticated.
    Redirects unauthenticated users to the "unauthorized" page.
    """
    template_name = "digits/home.html"
    form_class = ImageForm

    def form_valid(self, form: ImageForm):
        """
        This method is called when the form data is valid.
        Preprocesses the image and inputs it into the model for classification.
        Sets "digit_is_valid" to True and "digit" to the detected digit in the site context.
        :param form: The form that was validated.
        :return: Response
        """

        # load image
        f = form.cleaned_data["image"]
        img = preprocess_image(f.file)

        # input into model
        model = GACNN.get_trained()
        digit = model.classify_input(img)

        context_data = self.get_context_data(form=form)
        context_data["digit_is_valid"] = True
        context_data["digit"] = digit

        #messages.success(self.request, "You did it!")

        return render(self.request, self.template_name, context_data)
