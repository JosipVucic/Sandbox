# accounts/management/commands/list_users.py
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
User = get_user_model()


class Command(BaseCommand):
    help = 'List all users in the database'

    def handle(self, *args, **options):
        users = User.objects.all()
        for user in users:
            self.stdout.write(self.style.SUCCESS(f"{user.username} {user.email}"))
