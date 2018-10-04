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
    
  var mnemonic = '';
  
  chrome.storage.sync.get('mnemonic', function(result) {
    if (typeof(result.mnemonic) == 'undefined' || result.mnemonic.length == 0) {
      $('#message').addClass('error').text(err);
      return;
    } else {
      mnemonic = result.mnemonic;
    }          
    var entropy = mnemonicToEntropy(mnemonic, enwords);
    $('#mnemonic').val(mnemonic).addClass('disabled');
    $('#entropy').val(entropy).addClass('disabled');
  });
  
  $('#regenerate').click(function(e) {
    mnemonic = generateMnemonic(128, randomHexArray, enwords);
    chrome.storage.sync.set({'mnemonic': mnemonic});
    
    $('#message').html("A new mnemonic was generated");
    location.reload(true);
  });
  $('#clear').click(function(e) {
    chrome.storage.sync.set({'mnemonic': ''});
    $('#message').html("Mnemonic cleared");
  });
  $('#save').click(function(e) {
    chrome.storage.sync.set({'mnemonic': $('#mnemonic').val()});
    $('#message').html("Mnemonic saved");
  });
});
