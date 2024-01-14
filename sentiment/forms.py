from crispy_forms.helper import FormHelper
from django import forms


class ReviewForm(forms.Form):
    """The form used for image upload, uses a single ImageField labeled as "image"."""
    review = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'cols': 40}))

    def __init__(self, *args, **kwargs):
        super(ReviewForm, self).__init__(*args, **kwargs)

        # Create a FormHelper instance
        self.helper = FormHelper(self)
        self.helper.form_show_labels = False
        self.fields['review'].label = False