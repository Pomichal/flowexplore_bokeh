function table_to_csv(source, columns, num_of_columns) {
    // const columns = Object.keys(source.data);
    const nrows = source.get_length();
    // console.log(columns);
    // console.log(num_of_columns);

    const  cols = columns.split(" ");
    const lines = [cols.join(',')];


    // console.log(columns);
    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < num_of_columns; j++) {
            const column = cols[j];
            row.push(source.data[column][i].toString())
        }
        lines.push(row.join(','))
    }
    return lines.join('\n').concat('\n')
}


const filename = 'new_coordinates.csv';
const filename2 = 'populations.txt';
csv_file = table_to_csv(source, columns, num_of_columns);
const blob = new Blob([csv_file], { type: 'text/csv;charset=utf-8;' });
const blob_text = new Blob([text], { type: 'text/plain' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.target = '_blank';
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob_text, filename2)
} else {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob_text);
    link.download = filename2;
    link.target = '_blank';
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}