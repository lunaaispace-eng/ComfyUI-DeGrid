// addInstallButton(groupId, featureId, installId, buttonText)
// groupId is SwarmUI's cleaned form of the group name ("VAE DeGrid" -> vaedegrid, "VAE Enhance" -> vaeenhance).
// Both groups are served by the same node pack (this repo), so both buttons install the same feature.
addInstallButton('vaedegrid', 'degrid', 'degrid', 'Install VAE DeGrid');
addInstallButton('vaeenhance', 'degrid', 'degrid', 'Install VAE DeGrid node pack');
