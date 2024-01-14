from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import FormView
from .forms import ReviewForm

from .neural.model import SentimentModel
from .neural.preprocessing import clean_pipeline, preprocess_pipeline, pad_features


class HomeView(LoginRequiredMixin, FormView):
    """
    The home view, allows for image upload if the user is authenticated.
    Redirects unauthenticated users to the "unauthorized" page.
    """
    template_name = "sentiment/home.html"
    form_class = ReviewForm

    def form_valid(self, form: ReviewForm):
        """
        This method is called when the form data is valid.
        Preprocesses the review and inputs it into the model for classification.
        Sets "review_is_valid" to True and "sentiment" to the detected sentiment in the site context.
        :param form: The form that was validated.
        :return: Response
        """

        # load image
        review = form.cleaned_data["review"]
        review = clean_pipeline(review)
        review = preprocess_pipeline(review).split()

        # input into model
        model = SentimentModel.get_trained()
        review = [model.vocab(review)]
        review = pad_features(review, pad_id=model.vocab.get_stoi()['<PAD>'])
        sentiment = model.classify_input(review)

        context_data = self.get_context_data(form=form)
        context_data["review_is_valid"] = True
        context_data["sentiment"] = sentiment

        #messages.success(self.request, "You did it!")

        return render(self.request, self.template_name, context_data)
