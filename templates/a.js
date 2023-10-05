str='{"udit": true,"ajay_sir": false}'
while (1){
    var i=str.indexOf('"');
    i++;
    str=str.slice(i);
    i=str.indexOf('"');
    var res=str.substring(0,i);
    console.log(res)
    i++;
    i++;

    i++;
    str=str.slice(i);
    console.log(str)
    if(str[0]=='t'){
        // document.getElementById("there").innerHTML+="<tr><td>"+res+"</td><td>PRESENT</td></tr>";
        str=str.slice(4);
        console.log("<tr><td>"+res+"</td><td>PRESENT</td></tr>")

    }
    else{
        // document.getElementById("there").innerHTML+="<tr><td>"+res+"</td><td>ABSENT</td></tr>";
        str=str.slice(5);
        console.log("<tr><td>"+res+"</td><td>ABSENT</td></tr>")

    }

    if(str.length<7){
        break;
    }
}