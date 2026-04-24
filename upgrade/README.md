# Upgrade Steps

Using the Max78000 featherboard for our project, upgrading it will require us to upload Neural Network specific .c files to the microUSB manually and flashing it into the featherboard for our demo. Since we are compartmentalizing individual components and stringing them together through a OS (freeRTOS or zephyr or etc.), we can just flash it into the live performance OS and see immediate results for our demo.

As for our design document, we are hoping to implement a similar USB port to our final design to allow customers to perform service updates if they want to or upload new music models automatically. To consider our vintige community, these updates should be optional and backporting support (may not need to consider this in our design document).
