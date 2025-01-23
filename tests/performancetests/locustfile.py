from locust import HttpUser, between, task


# Do load testing using locust by running the following in the command line
# locust -f tests/performancetests/locustfile.py
# Give this URL to HOST in the locust application
# API_URL = "https://backend-8020-final-424957459314.europe-west1.run.app"

class MyUser(HttpUser):
    wait_time = between(1, 2)
    default_images_names = []
    image_to_use = None

    def on_start(self):
        """Load one of the default images on start"""
        response = self.client.get("/default_images/")
        if response.status_code == 200:
            self.default_images_names = response.json().get("images", [])
            if self.default_images_names:
                image_response = self.client.get(f"/default_images/{self.default_images_names[0]}")
                if image_response.status_code == 200:
                    self.image_to_use = image_response.content

    @task
    def predict(self):
        """Simulating the predict endpoint"""
        if self.image_to_use:
            self.client.post(f"/predict/", files={"file": self.image_to_use})
