from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired


class RegForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired()])
	submit = SubmitField('Submit', validators=[DataRequired()])
	airline = SelectField('Airline', choices=['Air India', 'Vistara', 'Jet Airways', 'Premium Economy flights'], validators=[DataRequired()])
	source = SelectField('Source Airport', choices=['Delhi','Mumbai','Kolkata','Chennai'], validators=[DataRequired()])
	