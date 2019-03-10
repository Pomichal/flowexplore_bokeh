var counter = 0;

function read_file(filename) {
    var reader = new FileReader();
    // reader.onload = load_handler;
    reader.onerror = error_handler;
    reader.onload = function(e) {
            // get file content
            let text = e.target.result;
            // console.log(text,filename.name);
            file_source.data = {'file_contents' : [text], 'file_name':[filename.name]};
            file_source.change.emit();
            counter += 1;
        };
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);

}

function error_handler(evt) {
    if(evt.target.error.name === "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.setAttribute('multiple','');

input.onchange = function(){
    if (window.FileReader) {
        for(let i=0; i < input.files.length; i++) {
            read_file(input.files[i]);
        }
        console.log(counter);
        alert('upload finished');
    } else {
        alert('FileReader is not supported in this browser');
    }
};
input.click();
