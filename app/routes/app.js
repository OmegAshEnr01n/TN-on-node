var express = require('express');
var router = express.Router();
var multer = require('multer');
var path = require('path');
var storage = multer.diskStorage({
    destination: function(req, file, cb){
        cb(null, 'public/files');
    }
});
var upload = multer({
    storage: storage
});
var shell = require('shelljs');
var cmd = require('node-cmd')
var fs = require('fs');
var base64Img = require('base64-img');
// I had problems with getting multer working. remeber to add name tag to anything that you need in the html.


router.get(
    '/',

    function (req, res) {


        res.render('app/index.ejs',{
            layout: 'app.ejs'
        });
    }
);

router
    .get(
        '/uploadimg',

        function(req, res){
            console.log();

            res.render('app/uploadimg.ejs',{
                layout: 'app.ejs'
            });
        }
    );

router
    .post(
        '/uploadimg',

        upload.any(),
        function(req, res){
            var temp = req.files[0];
            if(temp.fieldname != 'data'){
                res.redirect('/');
            } else {
                console.log('python '+path.join(__dirname,'..','py','eval.py')+' --cuda --load_model '+ path.join(__dirname,'..','py','300Gen_net.pt')+' --load_img '+path.join(__dirname,'..','public','files',temp.filename) +' --outfile '+path.join(__dirname,'..','public','files','temp.png'))
                shell.exec('python '+path.join(__dirname,'..','py','eval.py')+' --cuda --load_model '+ path.join(__dirname,'..','py','300Gen_net.pt')+' --load_img '+path.join(__dirname,'..','public','files',temp.filename) +' --outfile '+path.join(__dirname,'..','public','files','temp.png'))
                res.redirect('/app/modelresult?file='+temp.fieldname);
            }
        }
    )
;
router.get(
    '/modelresult',
    async function(req, res, next){

        var image = base64Img.base64Sync(path.join(__dirname,'..','public','files',req.query.file));
        var image2 = base64Img.base64Sync(path.join(__dirname,'..','public','files','temp.png'));
                    res.render('app/modelresult.ejs',{
                        layout: 'app.ejs',
                        image:  image,
                        image2: image2
                    });


    }
);


module.exports = router;
