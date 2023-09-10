import io
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import numpy as np
from PIL import Image


def send_notification(camera_index: int, frame: np.ndarray):
    sender = "Private Person <from@example.com>"
    receiver = "A Test User <to@example.com>"

    message = MIMEMultipart()
    message["Subject"] = "Suspicious activity on camera {}".format(camera_index)
    message["From"] = sender
    message["To"] = receiver
    message.preamble = "Preview in the attachment"
    outbuf = io.BytesIO()
    Image.fromarray(frame).save(outbuf, format="PNG")
    my_mime_image = MIMEImage(outbuf.getvalue())
    my_mime_image.add_header('Content-Disposition', 'attachment', filename='frame.png')
    outbuf.close()
    message.attach(my_mime_image)

    with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as server:
        server.login("e0faca00ac1a1e", "53b792d83abb1b")
        server.sendmail(sender, receiver, message.as_string())
