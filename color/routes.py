from color import app
from color.color_new import get_colors
from color.forms import PhotoForm
from flask import render_template, redirect, url_for, request, session
from werkzeug.utils import secure_filename
from webcolors import hex_to_rgb
import os
import uuid


@app.route('/', methods=['GET', 'POST'])
def index():
    form = PhotoForm()
    if form.validate_on_submit():
        print(hex_to_rgb(request.form.get('palette_outline_color')))
        f = form.photo.data
        filename = secure_filename(f.filename)
        _, ext = os.path.splitext(filename)
        filename = uuid.uuid4().hex + ext
        print(filename)
        f, pal2, hex_codes, pal3, hex_codes_compli, pal4a, hex_codes_triadic1, pal4b, hex_codes_triadic2, pal5a, hex_codes_tetradic1, pal5b, hex_codes_tetradic2, palml, hex_codes_ml = get_colors(f, palette_length_div=form.palette_height.data, outline_width=form.palette_outline_width.data,
                       outline_color=hex_to_rgb(request.form.get('palette_outline_color')))
        path = os.path.join(app.root_path, 'static/images',  filename)
        path2 = os.path.join(app.root_path, 'static/images', "pal"+filename)
        path3 = os.path.join(app.root_path, 'static/images', "pal3"+filename)
        path4a = os.path.join(app.root_path, 'static/images', "pal4a"+filename)
        path4b = os.path.join(app.root_path, 'static/images', "pal4b"+filename)
        path5a = os.path.join(app.root_path, 'static/images', "pal5a"+filename)
        path5b = os.path.join(app.root_path, 'static/images', "pal5b"+filename)
        pathml = os.path.join(app.root_path, 'static/images', "palml"+filename)

        session['hex_codes'] = hex_codes
        session['hex_codes_compli'] = hex_codes_compli
        session['hex_codes_triadic1'] = hex_codes_triadic1
        session['hex_codes_triadic2'] = hex_codes_triadic2
        session['hex_codes_tetradic1'] = hex_codes_tetradic1
        session['hex_codes_tetradic2'] = hex_codes_tetradic2
        session['hex_codes_ml'] = hex_codes_ml

        f.save(path)
        pal2.save(path2)
        pal3.save(path3)
        pal4a.save(path4a)
        pal4b.save(path4b)
        pal5a.save(path5a)
        pal5b.save(path5b)
        palml.save(pathml)

        return redirect(url_for('picture', name=filename, height=f.height, width=f.width))

    return render_template('index.html', form=form, src='default')


@app.route('/picture/<name>/<height>/<width>')
def picture(name, height, width):
    src = url_for('static', filename='images/' + name)
    src2 = url_for('static', filename='images/' + "pal" + name)
    src3 = url_for('static', filename='images/' + "pal3" + name)
    src4a = url_for('static', filename='images/' + "pal4a" + name)
    src4b = url_for('static', filename='images/' + "pal4b" + name)
    src5a = url_for('static', filename='images/' + "pal5a" + name)
    src5b = url_for('static', filename='images/' + "pal5b" + name)
    srcml = url_for('static', filename='images/' + "palml" + name)

    height, width = img_tag_size(int(height), int(width))
    return render_template('picture.html', src=src, src2=src2, src3=src3, src4a=src4a, src4b=src4b, src5a=src5a, src5b=src5b, srcml = srcml, height=height, width=width, 
        hex_codes=session.get('hex_codes'), hex_codes1=session.get('hex_codes_compli'), hex_codes2=session.get('hex_codes_triadic1'), hex_codes3=session.get('hex_codes_triadic2'), 
        hex_codes4=session.get('hex_codes_tetradic1'), hex_codes5=session.get('hex_codes_tetradic2'), hex_codes6=session.get('hex_codes_ml'))
    #return render_template('picture.html', src=src, src2=src2, height=height, width=width, hex_codes=session.get('hex_codes'))


def img_tag_size(height, width):
    if height < 500 and width < 500:
        return height, width
    else:
        while height > 500 and width > 500:
            height = int(height/2)
            width = int(width/2)
        return height, width


@app.errorhandler(413)
def error413(e):
    return render_template('413.html'), 413
