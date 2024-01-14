from crispy_forms.helper import FormHelper
from django import forms
from django.forms.fields import ImageField
from django.utils.translation import gettext_lazy as _


class ImageForm(forms.Form):
    """The form used for image upload, uses a single ImageField labeled as "image"."""
    image = ImageField()

    def __init__(self, *args, **kwargs):
        super(ImageForm, self).__init__(*args, **kwargs)

        # Create a FormHelper instance
        self.helper = FormHelper(self)
        self.helper.form_show_labels = False
        self.fields['image'].label = False
