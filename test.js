function hankaku2Zenkaku(str) {//全角を半角にする関数
  return str.replace(/[Ａ-Ｚａ-ｚ０-９]/g, function (s) {
    return String.fromCharCode(s.charCodeAt(0) - 0xFEE0);
  });
}

function sleep(ms) {//待機用の関数
  return new Promise(resolve => setTimeout(resolve, ms));//msたったらresolveを返す。
}


function array2csv(array) {//2次元配列専用CSV生成器
  //var num = array[0].length;
  var bom = new Uint8Array([0xEF, 0xBB, 0xBF]);//文字化け対策
  var csv_data = array.map(x => x.join(',')).join('\r\n');//2次元配列から文字列
  let blob = new Blob([bom, csv_data], { type: "csv/plain" });
  let link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'tukuttano.csv';
  link.click();

}



var main = async function () {
  var data_num = 101;//データの個数をコントロール
  var all_result = [];//最終的な結果を入れる二次元配列
  var all_company_num = codelist.length;//企業コードの数だけ。
  console.log(all_company_num)
  for (i = 3000; i < all_company_num; i++) {//メインループ all_company_numだけ繰り返す

    //var target = $('#target').val()
    var target = 'https://finance.yahoo.co.jp/quote/' + codelist[i] + '.T/history';
    //console.log('読み込み完了');
    console.log(i);
    $.ajax({
      url: `http://localhost/test.php?url=${target}`,
      type: 'GET',
      dataType: 'html',
      cache: false
    }).done(function (data) {
      console.log("ok");
      var result_array = [];//結果を格納する
      //$('#result').html($(data).find('._3rXWJKZF').text());
      var classname = '._3rXWJKZF';
      $(data).find(classname).each(function (i) {//i番目の処理　
        var tete = $(this).text();
        if ((i === 1 || i >= 5) && (i % 6 != 3)) {
          tete = tete.replace(',', '');
          result_array.push(Number(tete));
          //$('#result').append(tete,',');
        };
      })
      //console.log(result_array);
      if (result_array.length === data_num) {//データが規定の個数の時のみ処理を続ける
        all_result.push(result_array);
      };
      //console.log($(data).text());
    }).fail(function (XMLHttpRequest, textStatus, errorThrown) {
      console.log("error");
    })
    await sleep(1 * 1000);
  }
  console.log(all_result);
  array2csv(all_result);

}

$(document).ready(function () {

  $('.btn-search').on('click', main);//検索ボタンで処理の開始

  $('#target').keypress(function (e) {//エンターキーが押された時も処理を開始する
    if (e.which == 13) {
      console.log("key" + e.which);
      main();
      return false;
    };
  });
});
