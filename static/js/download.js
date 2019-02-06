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


const filename = 'data_result.csv';
filetext = table_to_csv(source, columns, num_of_columns);
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

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