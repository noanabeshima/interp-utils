import logging
import boto3
from botocore.exceptions import ClientError
import os


def upload_file(file_name, bucket, object_name=None, public=False):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        if public is True:
            s3_client.upload_file(
                file_name,
                bucket,
                object_name,
                ExtraArgs={
                    "ACL": "public-read",
                    "ContentDisposition": "inline",
                    "ContentType": "text/html",
                },
            )
        else:
            s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    url = f"https://{bucket}.s3.amazonaws.com/{object_name}"
    return url


html_head = """<head>
	<link rel="stylesheet"
          href="https://fonts.googleapis.com/css?family=Proxima_Nova">
	<style>
		body {
			font-family: 'Proxima_Nova', sans-serif;
		}
		p {
			font-size: 1.3rem;
		}
		h2 {
			font-size: 1.6rem
		}
	</style>
</head>

"""


def upload_figs(figs, fname, bucket="plotly-figs", fig_info=[], public=True):
    """Uploads a plotly figure to an S3 bucket
    Returns S3 url if upload is successful, otherwise returns False
    """
    if not isinstance(figs, list):
        figs = [figs]
    if not isinstance(fig_info, list):
        fig_info = [fig_info]
    if len(fig_info) == 0:
        fig_info = ["" for i in range(len(figs))]
    assert len(figs) == len(fig_info)

    # assert fname.split('.')[-1] == 'html', 'fname must end in .html'

    res_html = html_head
    for fig, info in zip(figs, fig_info):
        res_html += "<body>" + info + "</body>"
        res_html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    with open(f"/tmp/{fname}", "w") as f:
        f.write(res_html)
    url = upload_file("/tmp/" + fname, bucket, object_name=fname, public=public)
    os.remove("/tmp/" + fname)
    return url
