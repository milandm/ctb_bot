from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Create mock data for YourModel"

    def handle(self, *args, **kwargs):
        # Your code to create mock data here
        generate_affiliate()
        self.stdout.write(self.style.SUCCESS('Successfully created mock data'))



