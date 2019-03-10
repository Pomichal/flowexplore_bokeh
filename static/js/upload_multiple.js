function read_file(filename, file_list) {
    var reader = new FileReader();
    // reader.onload = load_handler;
    reader.onerror = error_handler;
    reader.onload = function(e) {
            // get file content
            let text = e.target.result;
            // console.log(text,filename.name);
            file_list.push({'file_contents' : [text], 'file_name':[filename.name]});
            // console.log(files.file_list);
            // let li = document.createElement("li");
            // li.innerHTML = name + "____" + text;
            // ul.appendChild(li);
            // files.change.emit();
        };
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

// function load_handler(event) {
//     var b64string = event.target.result;
//     console.log(event);
    // file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
    // file_source.change.emit();
// }

function error_handler(evt) {
    if(evt.target.error.name === "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.setAttribute('multiple','');
// files.file_list = [];
input.onchange = function(){
    let file_list = [];
    if (window.FileReader) {
        for(let i=0; i < input.files.length; i++) {
            read_file(input.files[i], file_list);
        }
        console.log(file_list);
        files.data = {'file_list': file_list};
        console.log(files);
        files.change.emit();
        console.log(files);
        files.change.emit();

    } else {
        alert('FileReader is not supported in this browser');
    }
};
input.click();
