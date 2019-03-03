const filename = 'populations.txt';

const blob_text = new Blob([text], { type: 'text/plain' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob_text, filename)
} else {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob_text);
    link.download = filename;
    link.target = '_blank';
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
