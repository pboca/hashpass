'use strict';

// Make sure we are in strict mode.
(function() {
  var strictMode = false;
  try {
    NaN = NaN;
  } catch (err) {
    strictMode = true;
  }
  if (!strictMode) {
    throw 'Unable to activate strict mode.';
  }
})();


$(function() {
  runTests();

  var saved_mnemonic = getCookie('mnemonic');
  if (saved_mnemonic !== null)
      $('#mnemonic').val(saved_mnemonic)

  var update = function() {
    // Compute the first 16 base64 characters of iterated-SHA-256(domain + '/' + key, 2 ^ difficulty).
    var key = $('#key').val();
    var username = $('#user').val();
    var domain = $('#domain').val();
    var mnemonic = $('#mnemonic').val();
    
    var entropy = null;
    try {
        entropy = mnemonicToEntropy(mnemonic, enwords);
    } catch (err) {
        $('#message').addClass('error').text("The mnemonic is invalid");
        return;
    }
    $('#message').removeClass('error').text("");
    
    setCookie('mnemonic', mnemonic, 30);

    var pwd = generateFoldedPassword(entropy, key, domain, username, '0');
    
    $('#hash').val(pwd);
    $('#entropy').val(entropy);
  };
  
  var timeout = null;
  var debouncedUpdate = function() {
    if (timeout !== null) {
      clearInterval(timeout);
    }
    timeout = setTimeout((function() {
      update();
      timeout = null;
    }), 100);
  };

  $('#domain, #key, #user').bind('propertychange change keyup input paste', debouncedUpdate);
});
