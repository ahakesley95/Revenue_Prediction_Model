'''
@author Alex Hakesley 16011419

error handling helper module 

contains handlers that are automatically called 
if certain exception is thrown e.g. HTTP 404
and generic error functions

'''

from flask import Blueprint, make_response

app_bp = Blueprint('error_handlers', __name__)

@app_bp.app_errorhandler(404)
def handle_not_found(e):
    return make_response({
        "error_help":"The specified endpoint does not exist."
    }), 404

@app_bp.app_errorhandler(405)
def error_method_not_allowed(e):
    return make_response({
        "error_help": "Endpoint does not support the specified method."
    }), 405

@app_bp.app_errorhandler(Exception)
def handle_internal_error(e):
    print(e)
    return {
        "error_help":"Contact the site administrator"
    }, 500

def handle_bad_request(error_message):
    return make_response({
        "error_help":error_message
    }), 400

def handle_unauthorized(error_message):
    return make_response({
        "error_help":error_message
    }), 401

