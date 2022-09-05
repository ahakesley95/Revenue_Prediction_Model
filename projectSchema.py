'''
@author Alex Hakesley 16011419

JSON validation schema module.

Includes classes of custom JSON fields and schema 
that are used to validate incoming JSON body. Return
some error if any of the body does not match the schema.

'''

from flask_marshmallow import fields, Schema, exceptions

class StrictBool(fields.fields.Field):
    def _deserialize(self, value, attr, obj, **kwargs):
        if isinstance(value, (bool)):
            return value
        raise exceptions.ValidationError("Not a valid boolean.")

class StrictFloat(fields.fields.Field):
    def _deserialize(self, value, attr, obj, **kwargs):
        if isinstance(value, (float, int)):
            return value
        raise exceptions.ValidationError("Not a valid float or integer.")

class PredictionDataSchema(Schema):
    ordered = True
    date = fields.fields.DateTime(
        required=True)
    open = StrictBool(required=True)
    sunrise = fields.fields.DateTime(
        format="%H:%M",
        required=True)
    sunset = fields.fields.DateTime(
        format="%H:%M",
        required=True)
    minTempC = StrictFloat(required=True)
    maxTempC = StrictFloat(required=True)

class TrainObjectDataSchema(Schema):
    ordered = True
    date = fields.fields.DateTime(
        required=True)
    open = StrictBool(required=True)
    revenue = StrictFloat(required=True)
    sunrise = fields.fields.DateTime(
        format="%H:%M",
        required=True)
    sunset = fields.fields.DateTime(
        format="%H:%M",
        required=True)
    minTempC = StrictFloat(required=True)
    maxTempC = StrictFloat(required=True)

class AuthenticationSchema(Schema):
    email = fields.fields.String(required=True)
    password = fields.fields.String(required=True)